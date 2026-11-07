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

"""Page pools made of one array per layer are sized from free HBM, each array within the largest free block."""

import math

import easydel.caching.mla_ragged_page.cache as mla_cache_mod
import easydel.caching.ragged_page.cache as ragged_cache_mod
import jax
import jax.numpy as jnp
import pytest
from easydel.infra.sharding import coerce_runtime_sharding_resolver
from spectrax import PartitionAxis

pytestmark = pytest.mark.cpu_ok

_LIMIT = 16 << 30
_UTIL = 0.9
_MARGIN = 256 << 20
_LAYERS = 8


class _FakeMesh:
    def __init__(self, shape):
        self.shape = shape


def _devices(monkeypatch, largest_free: int):
    class _Device:
        @staticmethod
        def memory_stats():
            return {"bytes_limit": _LIMIT, "bytes_in_use": 0, "largest_free_block_bytes": largest_free}

    monkeypatch.setattr(jax, "local_devices", lambda: [_Device()])


def _ragged():
    return ragged_cache_mod.RaggedPagesCacheConfig.create(
        mesh=_FakeMesh({"tp": 1}),
        runtime_sharding_resolver=coerce_runtime_sharding_resolver(
            PartitionAxis(kv_head_axis="tp", data_parallel_axis="dp")
        ),
        kvdtype=jnp.bfloat16,
        num_hidden_layers=_LAYERS,
        num_kv_heads=2,
        max_model_length=64,
        kv_head_dim_size=128,
        hbm_utilization=_UTIL,
        page_size=32,
        version="v3",
    )


def _mla():
    return mla_cache_mod.MLARaggedPagesCacheConfig.create(
        mesh=_FakeMesh({"tp": 1}),
        runtime_sharding_resolver=coerce_runtime_sharding_resolver(
            PartitionAxis(kv_head_axis="tp", data_parallel_axis="dp")
        ),
        kvdtype=jnp.bfloat16,
        num_hidden_layers=_LAYERS,
        num_kv_heads=1,
        max_model_length=64,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        hbm_utilization=_UTIL,
        page_size=32,
    )


def _layer_array_bytes(cfg) -> int:
    shape, _axes = cfg.get_shape_and_axes()
    return math.prod(shape) * jnp.dtype(cfg.kvdtype).itemsize


@pytest.mark.parametrize("build", [_ragged, _mla], ids=["ragged", "mla"])
def test_fragmented_hbm_fills_free_memory_with_each_layer_array_in_the_largest_block(monkeypatch, build):
    largest_free = 1 << 30  # 16 GiB free, but no free block larger than 1 GiB
    _devices(monkeypatch, largest_free)
    cfg = build()
    usable_block = int(largest_free * _UTIL) - _MARGIN
    # Every allocation is one layer's array, and each must fit in the largest free block.
    assert _layer_array_bytes(cfg) <= usable_block
    # The pool as a whole is not capped by that one block.
    assert _layer_array_bytes(cfg) * _LAYERS > usable_block * (_LAYERS - 1)


@pytest.mark.parametrize("build", [_ragged, _mla], ids=["ragged", "mla"])
def test_unfragmented_hbm_is_bounded_by_total_free_memory(monkeypatch, build):
    _devices(monkeypatch, largest_free=_LIMIT)
    cfg = build()
    usable_free = int(_LIMIT * _UTIL) - _MARGIN
    page_bytes = _layer_array_bytes(cfg) // cfg.num_pages * _LAYERS
    assert cfg.num_pages == usable_free // page_bytes


def test_nearly_unfragmented_hbm_keeps_the_previous_pool_bound(monkeypatch):
    # The 16-chip serving case: the largest block is about 87% of free HBM. That is not real fragmentation
    # for a per-layer pool, so its page count and compiled shapes must stay exactly as before the fix.
    largest_free = int(_LIMIT * 0.87)
    _devices(monkeypatch, largest_free)
    cfg = _ragged()
    old_budget = min(int(_LIMIT * _UTIL) - _MARGIN, int(largest_free * _UTIL) - _MARGIN)
    page_bytes = _layer_array_bytes(cfg) // cfg.num_pages * _LAYERS
    assert cfg.num_pages == old_budget // page_bytes


def test_a_single_array_pool_keeps_the_largest_block_bound(monkeypatch):
    _devices(monkeypatch, largest_free=1 << 30)
    assert ragged_cache_mod.per_device_hbm_budget_bytes(_UTIL, mode="free") == int((1 << 30) * _UTIL) - _MARGIN
    with pytest.raises(ValueError, match="largest_allocation_share"):
        ragged_cache_mod.per_device_hbm_budget_bytes(_UTIL, mode="free", largest_allocation_share=0.0)
