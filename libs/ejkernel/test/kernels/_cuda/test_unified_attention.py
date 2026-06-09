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
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ejkernel.kernels._cuda.unified_attention import unified_attention as cuda_unified_attention
from ejkernel.kernels._xla.unified_attention import unified_attention as xla_unified_attention

pytestmark = pytest.mark.skipif(
    jax.devices()[0].platform != "gpu" or shutil.which("nvcc") is None,
    reason="CUDA unified_attention tests require GPU backend and nvcc",
)

try:
    from ejkernel.kernels._cuda.unified_attention._build import build_cuda_lib

    build_cuda_lib()
except RuntimeError as exc:
    pytest.skip(f"CUDA unified_attention build failed: {exc}", allow_module_level=True)


def _make_inputs(
    *,
    rng_seed: int = 0,
    num_seqs: int = 3,
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


def test_unified_attention_cuda_matches_xla_basic():
    batch = _make_inputs(
        rng_seed=0,
        num_seqs=2,
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=64,
        kv_lens=[16, 12],
        q_lens=[16, 12],
        dtype=jnp.bfloat16,
    )
    scale = 1.0 / math.sqrt(batch["queries"].shape[-1])

    dev = jax.devices("gpu")[0]
    batch = {k: jax.device_put(v, dev) for k, v in batch.items()}

    out_cuda = cuda_unified_attention(
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

    out_cuda = jax.block_until_ready(out_cuda)
    out_xla = jax.block_until_ready(out_xla)

    np.testing.assert_allclose(
        np.asarray(out_cuda, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )


def test_unified_attention_cuda_matches_xla_with_sinks():
    batch = _make_inputs(
        rng_seed=1,
        num_seqs=3,
        num_q_heads=8,
        num_kv_heads=4,
        head_dim=64,
        kv_lens=[16, 16, 16],
        q_lens=[16, 16, 16],
        dtype=jnp.float16,
    )
    scale = 1.0 / math.sqrt(batch["queries"].shape[-1])

    num_q_heads = batch["queries"].shape[1]
    key = jax.random.PRNGKey(123)
    softmax_aux = jax.random.normal(key, (num_q_heads,), dtype=jnp.float32)

    dev = jax.devices("gpu")[0]
    batch = {k: jax.device_put(v, dev) for k, v in batch.items()}
    softmax_aux = jax.device_put(softmax_aux, dev)

    out_cuda = cuda_unified_attention(
        batch["queries"],
        batch["key_cache"],
        batch["value_cache"],
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        softmax_scale=scale,
        softmax_aux=softmax_aux,
    )
    out_xla = xla_unified_attention(
        batch["queries"],
        batch["key_cache"],
        batch["value_cache"],
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        softmax_scale=scale,
        softmax_aux=softmax_aux,
    )

    out_cuda = jax.block_until_ready(out_cuda)
    out_xla = jax.block_until_ready(out_xla)

    np.testing.assert_allclose(
        np.asarray(out_cuda, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=2e-2,
        atol=5e-2,
    )


def test_unified_attention_cuda_matches_xla_sliding_window():
    batch = _make_inputs(
        rng_seed=2,
        num_seqs=2,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=32,
        kv_lens=[24, 20],
        q_lens=[24, 20],
        dtype=jnp.float16,
    )
    scale = 1.0 / math.sqrt(batch["queries"].shape[-1])

    dev = jax.devices("gpu")[0]
    batch = {k: jax.device_put(v, dev) for k, v in batch.items()}

    out_cuda = cuda_unified_attention(
        batch["queries"],
        batch["key_cache"],
        batch["value_cache"],
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        softmax_scale=scale,
        sliding_window=8,
    )
    out_xla = xla_unified_attention(
        batch["queries"],
        batch["key_cache"],
        batch["value_cache"],
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        softmax_scale=scale,
        sliding_window=8,
    )

    out_cuda = jax.block_until_ready(out_cuda)
    out_xla = jax.block_until_ready(out_xla)

    np.testing.assert_allclose(
        np.asarray(out_cuda, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
