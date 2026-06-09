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

from ejkernel.kernels._cuda.ragged_page_attention_v3 import (
    ragged_page_attention_v3 as cuda_ragged_page_attention_v3,
)
from ejkernel.kernels._xla.ragged_page_attention_v3 import (
    ragged_page_attention_v3 as xla_ragged_page_attention_v3,
)
from ejkernel.utils import make_dummy_rpa_inputs

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="CUDA tests require GPU backend")


def _to_device(batch, dev):
    out = {}
    for k, v in batch.items():
        if isinstance(v, (jax.Array, np.ndarray)):
            out[k] = jax.device_put(v, dev)
        else:
            out[k] = v
    return out


def test_ragged_page_attention_v3_cuda_matches_xla():
    batch = make_dummy_rpa_inputs(
        rng_seed=0,
        num_seqs=2,
        pages_per_seq=2,
        page_size=16,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=64,
        kv_dtype=jnp.float16,
        total_q=8,
    )

    dev = jax.devices("gpu")[0]
    batch = _to_device(batch, dev)

    kv_cache_xla = batch["kv_cache"].copy()
    kv_cache_cuda = batch["kv_cache"].copy()

    sinks = jnp.zeros((batch["queries"].shape[1],), dtype=jnp.float32)
    sinks = jax.device_put(sinks, dev)

    out_xla, cache_xla = xla_ragged_page_attention_v3(
        batch["queries"],
        batch["keys"],
        batch["values"],
        kv_cache_xla,
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        sinks,
        softmax_scale=batch["queries"].shape[-1] ** -0.5,
        sliding_window=16,
        logits_soft_cap=10.0,
    )
    out_cuda, cache_cuda = cuda_ragged_page_attention_v3(
        batch["queries"],
        batch["keys"],
        batch["values"],
        kv_cache_cuda,
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        sinks,
        softmax_scale=batch["queries"].shape[-1] ** -0.5,
        sliding_window=16,
        logits_soft_cap=10.0,
    )

    out_cuda = jax.block_until_ready(out_cuda)
    out_xla = jax.block_until_ready(out_xla)
    cache_cuda = jax.block_until_ready(cache_cuda)
    cache_xla = jax.block_until_ready(cache_xla)

    np.testing.assert_allclose(
        np.asarray(out_cuda, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    assert cache_cuda.shape == cache_xla.shape
    assert cache_cuda.dtype == cache_xla.dtype


def test_ragged_page_attention_v3_cuda_matches_xla_with_scales():
    batch = make_dummy_rpa_inputs(
        rng_seed=1,
        num_seqs=3,
        pages_per_seq=2,
        page_size=16,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=64,
        kv_dtype=jnp.bfloat16,
        total_q=12,
    )

    dev = jax.devices("gpu")[0]
    batch = _to_device(batch, dev)

    kv_cache_xla = batch["kv_cache"].copy()
    kv_cache_cuda = batch["kv_cache"].copy()

    sinks = jnp.zeros((batch["queries"].shape[1],), dtype=jnp.float32)
    sinks = jax.device_put(sinks, dev)

    out_xla, _cache_xla = xla_ragged_page_attention_v3(
        batch["queries"],
        batch["keys"],
        batch["values"],
        kv_cache_xla,
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        sinks,
        softmax_scale=batch["queries"].shape[-1] ** -0.5,
        sliding_window=None,
        logits_soft_cap=None,
        q_scale=0.9,
        k_scale=1.1,
        v_scale=0.8,
    )
    out_cuda, _cache_cuda = cuda_ragged_page_attention_v3(
        batch["queries"],
        batch["keys"],
        batch["values"],
        kv_cache_cuda,
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        sinks,
        softmax_scale=batch["queries"].shape[-1] ** -0.5,
        sliding_window=None,
        logits_soft_cap=None,
        q_scale=0.9,
        k_scale=1.1,
        v_scale=0.8,
    )

    out_cuda = jax.block_until_ready(out_cuda)
    out_xla = jax.block_until_ready(out_xla)

    np.testing.assert_allclose(
        np.asarray(out_cuda, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
