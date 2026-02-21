import jax.numpy as jnp
from eformer.escale import PartitionAxis, PartitionManager

import easydel.caching.ragged_page.cache as ragged_cache_mod
import easydel.caching.unified_attention.cache as unified_cache_mod


class _FakeMesh:
    def __init__(self, shape):
        self.shape = shape


def _mesh_tp_only():
    return _FakeMesh({"tp": 1})


def _mesh_dp_tp():
    return _FakeMesh({"dp": 2, "tp": 1})


def _partition_manager():
    return PartitionManager(PartitionAxis(kv_head_axis="tp", data_parallel_axis="dp"))


def test_ragged_page_budget_scales_with_mesh_dp_axis(monkeypatch):
    monkeypatch.setattr(
        ragged_cache_mod,
        "per_device_hbm_budget_bytes",
        lambda *_args, **_kwargs: 1 << 20,
    )

    pm = _partition_manager()
    cfg_no_dp = ragged_cache_mod.RaggedPagesCacheConfig.create(
        mesh=_mesh_tp_only(),
        partition_manager=pm,
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
        partition_manager=pm,
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


def test_unified_page_budget_scales_with_mesh_dp_axis(monkeypatch):
    monkeypatch.setattr(
        unified_cache_mod,
        "per_device_hbm_budget_bytes",
        lambda *_args, **_kwargs: 1 << 20,
    )

    pm = _partition_manager()
    cfg_no_dp = unified_cache_mod.UnifiedAttentionCacheConfig.create(
        mesh=_mesh_tp_only(),
        partition_manager=pm,
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
        partition_manager=pm,
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
