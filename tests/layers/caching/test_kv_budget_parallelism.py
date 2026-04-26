# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

import jax.numpy as jnp
from spectrax import PartitionAxis

import easydel.caching.ragged_page.cache as ragged_cache_mod
import easydel.caching.unified_attention.cache as unified_cache_mod
from easydel.infra.sharding import coerce_runtime_sharding_resolver


class _FakeMesh:
    def __init__(self, shape):
        self.shape = shape


def _mesh_tp_only():
    return _FakeMesh({"tp": 1})


def _mesh_dp_tp():
    return _FakeMesh({"dp": 2, "tp": 1})


def _partition_manager():
    return coerce_runtime_sharding_resolver(PartitionAxis(kv_head_axis="tp", data_parallel_axis="dp"))


def test_ragged_page_budget_scales_with_mesh_dp_axis(monkeypatch):
    monkeypatch.setattr(
        ragged_cache_mod,
        "per_device_hbm_budget_bytes",
        lambda *_args, **_kwargs: 1 << 20,
    )

    pm = _partition_manager()
    cfg_no_dp = ragged_cache_mod.RaggedPagesCacheConfig.create(
        mesh=_mesh_tp_only(),
        runtime_sharding_resolver=pm,
        kvdtype=jnp.float32,
        num_hidden_layers=1,
        num_kv_heads=1,
        max_model_length=32,
        kv_head_dim_size=1,
        hbm_utilization=0.9,
        page_size=1,
    )
    cfg_dp2 = ragged_cache_mod.RaggedPagesCacheConfig.create(
        mesh=_mesh_dp_tp(),
        runtime_sharding_resolver=pm,
        kvdtype=jnp.float32,
        num_hidden_layers=1,
        num_kv_heads=1,
        max_model_length=32,
        kv_head_dim_size=1,
        hbm_utilization=0.9,
        page_size=1,
    )

    assert cfg_no_dp.data_parallel_size == 1
    assert cfg_dp2.data_parallel_size == 2
    assert cfg_dp2.num_pages == cfg_no_dp.num_pages * 2


def test_ragged_page_budget_replicates_kv_cache_when_tp_head_sharding_is_incompatible(monkeypatch):
    monkeypatch.setattr(
        ragged_cache_mod,
        "per_device_hbm_budget_bytes",
        lambda *_args, **_kwargs: 1 << 20,
    )

    pm = _partition_manager()
    cfg = ragged_cache_mod.RaggedPagesCacheConfig.create(
        mesh=_FakeMesh({"dp": 1, "tp": 4}),
        runtime_sharding_resolver=pm,
        kvdtype=jnp.bfloat16,
        num_hidden_layers=1,
        num_kv_heads=1,
        max_model_length=32,
        kv_head_dim_size=256,
        hbm_utilization=0.9,
        page_size=1,
        version="v3",
    )

    _shape, axes = cfg.get_shape_and_axes()

    assert cfg.kv_head_shards == 1
    assert axes[2] == ragged_cache_mod.common_types.EMPTY


def test_ragged_v3_storage_keeps_combined_kv_heads_for_small_head_dim(monkeypatch):
    monkeypatch.setattr(
        ragged_cache_mod,
        "per_device_hbm_budget_bytes",
        lambda *_args, **_kwargs: 1 << 20,
    )

    cfg = ragged_cache_mod.RaggedPagesCacheConfig.create(
        mesh=_mesh_tp_only(),
        runtime_sharding_resolver=_partition_manager(),
        kvdtype=jnp.bfloat16,
        num_hidden_layers=1,
        num_kv_heads=2,
        max_model_length=160,
        kv_head_dim_size=64,
        hbm_utilization=0.9,
        page_size=32,
        version="v3",
    )

    shape, _axes = cfg.get_shape_and_axes()

    assert shape[2] * shape[3] >= cfg.num_kv_heads * 2
    assert shape[4] >= cfg.k_headdim


def test_unified_page_budget_scales_with_mesh_dp_axis(monkeypatch):
    monkeypatch.setattr(
        unified_cache_mod,
        "per_device_hbm_budget_bytes",
        lambda *_args, **_kwargs: 1 << 20,
    )

    pm = _partition_manager()
    cfg_no_dp = unified_cache_mod.UnifiedAttentionCacheConfig.create(
        mesh=_mesh_tp_only(),
        runtime_sharding_resolver=pm,
        kvdtype=jnp.float32,
        num_hidden_layers=1,
        num_kv_heads=1,
        max_model_length=32,
        head_dim=1,
        hbm_utilization=0.9,
        page_size=1,
    )
    cfg_dp2 = unified_cache_mod.UnifiedAttentionCacheConfig.create(
        mesh=_mesh_dp_tp(),
        runtime_sharding_resolver=pm,
        kvdtype=jnp.float32,
        num_hidden_layers=1,
        num_kv_heads=1,
        max_model_length=32,
        head_dim=1,
        hbm_utilization=0.9,
        page_size=1,
    )

    assert cfg_no_dp.data_parallel_size == 1
    assert cfg_dp2.data_parallel_size == 2
    assert cfg_dp2.num_pages == cfg_no_dp.num_pages * 2
