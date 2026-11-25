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

"""Module-operation, registry, and shard_map tests for DualStreamRaggedPageAttention."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._registry import Backend, Platform, kernel_registry
from ejkernel.kernels._xla.ragged_page_attention_v3._xla_impl_fwd import (
    align_to,
    get_dtype_packing,
    merge_kv,
)
from ejkernel.modules import DualStreamRaggedPageAttention, dual_stream_ragged_page_attention
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as Ps


def _build_problem(seed: int = 0, *, head_dim: int = 128, num_q_heads: int = 8, num_kv_heads: int = 4):
    dtype = jnp.bfloat16
    page_size = 16
    seq_len = 24
    key = jax.random.PRNGKey(seed)
    kq, kkg, kvg, kkl, kvl = jax.random.split(key, 5)
    q = jax.random.normal(kq, (seq_len, num_q_heads, head_dim), dtype=dtype) * 0.25
    kg = jax.random.normal(kkg, (seq_len, num_kv_heads, head_dim), dtype=dtype) * 0.25
    vg = jax.random.normal(kvg, (seq_len, num_kv_heads, head_dim), dtype=dtype) * 0.25
    kl = jax.random.normal(kkl, (seq_len, num_kv_heads, head_dim), dtype=dtype) * 0.25
    vl = jax.random.normal(kvl, (seq_len, num_kv_heads, head_dim), dtype=dtype) * 0.25

    packing = get_dtype_packing(dtype)
    head_groups = align_to(num_kv_heads * 2, packing) // packing
    hd_pad = align_to(head_dim, 128)
    g_cache = jnp.zeros((4, page_size, head_groups, packing, hd_pad), dtype=dtype)
    l_cache = jnp.zeros((4, page_size, head_groups, packing, hd_pad), dtype=dtype)
    merged_g = merge_kv(kg, vg)
    g_cache = g_cache.at[1].set(merged_g[:page_size])
    g_cache = g_cache.at[2, : seq_len - page_size].set(merged_g[page_size:])
    lens = jnp.asarray([seq_len], dtype=jnp.int32)
    bt = jnp.asarray([1, 2], dtype=jnp.int32)
    qsl = jnp.asarray([0, seq_len], dtype=jnp.int32)
    dist = jnp.asarray([0, 0, 1], dtype=jnp.int32)
    return q, kl, vl, g_cache, l_cache, lens, bt, qsl, dist


def test_registry_signatures_and_structure_gate():
    """Verify registry signature compatibility, backend lookups, and shard_map wrapper structure."""
    assert "create_shard_map_wrapper" in DualStreamRaggedPageAttention.__dict__
    kernel_registry.validate_signatures("dual_stream_ragged_page_attention")
    xla_impl = kernel_registry.get("dual_stream_ragged_page_attention", platform=Platform.XLA, backend=Backend.ANY)
    assert callable(xla_impl)
    tpu_impl = kernel_registry.get("dual_stream_ragged_page_attention", platform=Platform.PALLAS, backend=Backend.TPU)
    assert callable(tpu_impl)


@pytest.mark.parametrize("overlap", [True, False])
def test_module_dual_stream_rpa_auto_matches_xla(overlap):
    """Public ejkernel.modules.dual_stream_ragged_page_attention auto dispatch matches XLA."""
    q, kl, vl, g_cache, l_cache, lens, bt, qsl, dist = _build_problem(seed=11 + int(overlap))
    out_auto, lc_auto = dual_stream_ragged_page_attention(
        q, kl, vl, g_cache, jnp.array(l_cache, copy=True), lens, bt, lens, bt, qsl, dist,
        sliding_window=12,
        overlap=overlap,
        platform="auto",
    )
    out_xla, lc_xla = dual_stream_ragged_page_attention(
        q, kl, vl, g_cache, jnp.array(l_cache, copy=True), lens, bt, lens, bt, qsl, dist,
        sliding_window=12,
        overlap=overlap,
        platform="xla",
    )
    np.testing.assert_allclose(np.asarray(out_auto, np.float32), np.asarray(out_xla, np.float32), atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(np.asarray(lc_auto, np.float32), np.asarray(lc_xla, np.float32), atol=1e-5)


def test_module_dual_stream_rpa_shard_map_matches_unsharded():
    """Head-sharded executor shard_map execution matches unsharded output."""
    devices = jax.devices()
    tp = min(len(devices), 4)
    if tp < 2:
        pytest.skip("Need >= 2 accelerator devices to test head-sharded shard_map.")

    q, kl, vl, g_cache, l_cache, lens, bt, qsl, dist = _build_problem(
        seed=77,
        head_dim=128,
        num_q_heads=8,
        num_kv_heads=tp,
    )
    out_ref, lc_ref = dual_stream_ragged_page_attention(
        q, kl, vl, g_cache, jnp.array(l_cache, copy=True), lens, bt, lens, bt, qsl, dist,
        sliding_window=12,
        overlap=True,
        platform="auto",
    )

    mesh = Mesh(np.asarray(devices[:tp]), ("tp",))
    q_spec = Ps(None, "tp", None)
    kv_spec = Ps(None, "tp", None)
    pages_spec = Ps(None, None, "tp", None, None)
    out_sharded, lc_sharded = dual_stream_ragged_page_attention(
        q, kl, vl, g_cache, jnp.array(l_cache, copy=True), lens, bt, lens, bt, qsl, dist,
        sliding_window=12,
        overlap=True,
        platform="auto",
        mesh=mesh,
        in_specs=(q_spec, kv_spec, kv_spec, pages_spec, pages_spec, Ps(), Ps(), Ps(), Ps(), Ps(), Ps(), None),
        out_specs=(q_spec, pages_spec),
    )
    np.testing.assert_allclose(
        np.asarray(out_sharded, np.float32),
        np.asarray(out_ref, np.float32),
        atol=2e-2,
        rtol=2e-2,
    )
    np.testing.assert_allclose(
        np.asarray(lc_sharded, np.float32),
        np.asarray(lc_ref, np.float32),
        atol=1e-5,
    )
