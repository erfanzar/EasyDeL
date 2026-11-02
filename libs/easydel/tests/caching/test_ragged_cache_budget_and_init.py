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

"""Regression tests: ragged v3 page budget, ``init_empty`` contract, shared HBM budget helper."""

import threading
import time
from types import SimpleNamespace

import easydel.caching.mla_ragged_page.cache as mla_cache_mod
import easydel.caching.ragged_page.cache as ragged_cache_mod
import easydel.caching.unified_attention.cache as unified_cache_mod
import jax
import jax.numpy as jnp
import pytest
from easydel.caching import MLARaggedPagesCache, RaggedPagesCache, TransformerCache, TurboQuantRaggedPagesCache
from easydel.infra.sharding import coerce_runtime_sharding_resolver
from spectrax import PartitionAxis

pytestmark = pytest.mark.cpu_ok


class _FakeMesh:
    def __init__(self, shape):
        self.shape = shape


def _resolver():
    return coerce_runtime_sharding_resolver(PartitionAxis(kv_head_axis="tp", data_parallel_axis="dp"))


def _create(monkeypatch, *, head_dim: int, num_kv_heads: int, budget: int):
    monkeypatch.setattr(ragged_cache_mod, "per_device_hbm_budget_bytes", lambda *_a, **_k: budget)
    return ragged_cache_mod.RaggedPagesCacheConfig.create(
        mesh=_FakeMesh({"tp": 1}),
        runtime_sharding_resolver=_resolver(),
        kvdtype=jnp.bfloat16,
        num_hidden_layers=1,
        num_kv_heads=num_kv_heads,
        max_model_length=64,
        kv_head_dim_size=head_dim,
        hbm_utilization=0.9,
        page_size=32,
        version="v3",
    )


def test_v3_head64_page_pool_is_not_halved(monkeypatch):
    # head_dim 64: K/V share the 128-wide head-dim axis, so one page costs
    # page_size * align(num_kv_heads, packing) * 128 * 2 bytes (bf16) = 32 * 2 * 128 * 2.
    cfg = _create(monkeypatch, head_dim=64, num_kv_heads=2, budget=1 << 24)
    assert cfg.num_pages == (1 << 24) // (32 * 2 * 128 * 2)


def test_v3_head128_page_pool_counts_key_and_value_heads(monkeypatch):
    # head_dim 128: the pair is interleaved on the head axis -> align(2 * 2, packing) heads.
    cfg = _create(monkeypatch, head_dim=128, num_kv_heads=2, budget=1 << 24)
    assert cfg.num_pages == (1 << 24) // (32 * 4 * 128 * 2)


@pytest.mark.parametrize("num_kv_heads", [1, 2, 3])
def test_v3_budget_matches_allocated_storage_bytes(monkeypatch, num_kv_heads):
    for head_dim in (64, 128):
        cfg = _create(monkeypatch, head_dim=head_dim, num_kv_heads=num_kv_heads, budget=1 << 26)
        shape, _axes = cfg.get_shape_and_axes()
        per_page_elems = 1
        for dim in shape[1:]:
            per_page_elems *= dim
        assert cfg.num_pages == (1 << 26) // (per_page_elems * 2)


@pytest.mark.parametrize("cls", [RaggedPagesCache, MLARaggedPagesCache, TurboQuantRaggedPagesCache])
def test_ragged_init_empty_is_classmethod_returning_placeholder_cache(cls):
    cache = cls.init_empty(3)
    assert isinstance(cache, cls)
    assert cache.views == [None, None, None]


def test_init_empty_accepts_model_config_for_ragged_and_transformer_caches():
    config = SimpleNamespace(num_hidden_layers=4)
    assert RaggedPagesCache.init_empty(config).views == [None] * 4
    assert TransformerCache.init_empty(config).views == [None] * 4


def test_unified_cache_budget_applies_largest_free_block_clamp(monkeypatch):
    class _Device:
        @staticmethod
        def memory_stats():
            return {"bytes_limit": 10 << 30, "bytes_in_use": 0, "largest_free_block_bytes": 1 << 30}

    monkeypatch.setattr(jax, "local_devices", lambda: [_Device()])
    expected = int((1 << 30) * 0.9) - (256 << 20)
    assert unified_cache_mod.per_device_hbm_budget_bytes(0.9, mode="free") == expected
    assert ragged_cache_mod.per_device_hbm_budget_bytes(0.9, mode="free") == expected


@pytest.mark.parametrize("page_bytes, expected", [(1, 64), (1 << 20, 16), (16 << 20, 1), (32 << 20, 1)])
def test_unified_update_slice_count_uses_shared_window_policy(monkeypatch, page_bytes, expected):
    monkeypatch.setattr(unified_cache_mod, "per_device_hbm_budget_bytes", lambda *_a, **_k: 1 << 40)
    cfg = unified_cache_mod.UnifiedAttentionCacheConfig.create(
        mesh=_FakeMesh({"tp": 1}),
        runtime_sharding_resolver=_resolver(),
        kvdtype=jnp.float32,
        num_hidden_layers=1,
        num_kv_heads=1,
        max_model_length=32,
        head_dim=max(1, page_bytes // 8),
        page_size=1,
    )
    assert cfg.num_slices_per_kv_cache_update_page == expected


def test_ragged_slice_window_keeps_strict_oversized_page_validation():
    with pytest.raises(ValueError):
        ragged_cache_mod.get_num_slices_per_kv_cache_update_page(32 << 20)


class _FakeKeyValueStore:
    """In-memory stand-in for the ``jax.distributed`` key-value client (thread-safe, blocking gets)."""

    def __init__(self):
        self.values: dict[str, str] = {}
        self._cond = threading.Condition()

    def key_value_set(self, key, value, allow_overwrite=False):
        with self._cond:
            if key in self.values and not allow_overwrite:
                raise RuntimeError(f"ALREADY_EXISTS: {key}")
            self.values[key] = value
            self._cond.notify_all()

    def blocking_key_value_get(self, key, timeout_in_ms):
        deadline = time.monotonic() + timeout_in_ms / 1000
        with self._cond:
            while key not in self.values:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"DEADLINE_EXCEEDED: {key}")
                self._cond.wait(remaining)
            return self.values[key]


def test_cache_budget_is_the_minimum_over_processes(monkeypatch):
    # Hosts of one JAX world see different free HBM after loading weights. Each host must size its KV pool from
    # the smallest budget of the round, or the hosts compile differently shaped step programs.
    from jax._src import distributed as jax_distributed

    free_gib = {0: (8, 5), 1: (6, 9), 2: (7, 7)}  # per process: free HBM in the first and the second round
    local = threading.local()

    class _Device:
        @staticmethod
        def memory_stats():
            return {"bytes_limit": 16 << 30, "bytes_in_use": (16 - free_gib[local.rank][local.round]) << 30}

    monkeypatch.setattr(jax_distributed.global_state, "client", _FakeKeyValueStore())
    monkeypatch.setattr(jax, "process_count", lambda: len(free_gib))
    monkeypatch.setattr(jax, "process_index", lambda: local.rank)
    monkeypatch.setattr(jax, "local_devices", lambda: [_Device()])
    # The cache's mesh holds a device of every process, so all of them take part in each round.
    spanning_mesh = SimpleNamespace(devices=[[SimpleNamespace(process_index=rank)] for rank in free_gib])
    budgets: dict[int, list[int]] = {}

    def _host(rank):
        local.rank = rank
        budgets[rank] = []
        for round_index in range(2):
            local.round = round_index
            budgets[rank].append(ragged_cache_mod.per_device_hbm_budget_bytes(0.5, mode="free", mesh=spanning_mesh))

    threads = [threading.Thread(target=_host, args=(rank,), daemon=True) for rank in free_gib]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads)
    expected = [int((gib << 30) * 0.5) - (256 << 20) for gib in (6, 5)]
    assert budgets == {rank: expected for rank in free_gib}


def test_cache_budget_on_a_host_local_mesh_needs_no_agreement(monkeypatch):
    # A cache whose mesh holds only this process's devices must not wait for the rest of the JAX world.
    from jax._src import distributed as jax_distributed

    class _StoreThatMustNotBeUsed:
        def key_value_set(self, *_args, **_kwargs):
            raise AssertionError("a host-local cache budget must not use the key-value store")

        def blocking_key_value_get(self, *_args, **_kwargs):
            raise AssertionError("a host-local cache budget must not use the key-value store")

    class _Device:
        process_index = 1

        @staticmethod
        def memory_stats():
            return {"bytes_limit": 16 << 30, "bytes_in_use": 6 << 30}

    monkeypatch.setattr(jax_distributed.global_state, "client", _StoreThatMustNotBeUsed())
    monkeypatch.setattr(jax, "process_count", lambda: 4)
    monkeypatch.setattr(jax, "process_index", lambda: 1)
    monkeypatch.setattr(jax, "local_devices", lambda: [_Device()])
    host_local_mesh = SimpleNamespace(devices=[[_Device(), _Device()]])
    budget = ragged_cache_mod.per_device_hbm_budget_bytes(0.5, mode="free", mesh=host_local_mesh)
    assert budget == int((10 << 30) * 0.5) - (256 << 20)


class _UnusedKeyValueStore:
    """A ``jax.distributed`` client stand-in that fails the test when a budget query touches it."""

    def key_value_set(self, *_args, **_kwargs):
        raise AssertionError("this cache budget must not use the key-value store")

    def blocking_key_value_get(self, *_args, **_kwargs):
        raise AssertionError("this cache budget must not use the key-value store")


def test_cache_budget_without_a_mesh_keeps_the_local_value(monkeypatch):
    # A caller that passes no mesh may run on only some processes of the JAX world: it must never wait for the rest.
    from jax._src import distributed as jax_distributed

    class _Device:
        @staticmethod
        def memory_stats():
            return {"bytes_limit": 16 << 30, "bytes_in_use": 6 << 30}

    monkeypatch.setattr(jax_distributed.global_state, "client", _UnusedKeyValueStore())
    monkeypatch.setattr(jax, "process_count", lambda: 4)
    monkeypatch.setattr(jax, "process_index", lambda: 2)
    monkeypatch.setattr(jax, "local_devices", lambda: [_Device()])
    assert ragged_cache_mod.per_device_hbm_budget_bytes(0.5, mode="free") == int((10 << 30) * 0.5) - (256 << 20)


def _ragged_config(mesh):
    return ragged_cache_mod.RaggedPagesCacheConfig.create(
        mesh=mesh,
        runtime_sharding_resolver=_resolver(),
        kvdtype=jnp.bfloat16,
        num_hidden_layers=1,
        num_kv_heads=2,
        max_model_length=64,
        kv_head_dim_size=128,
        hbm_utilization=0.9,
        page_size=32,
        version="v3",
    )


def _unified_config(mesh):
    return unified_cache_mod.UnifiedAttentionCacheConfig.create(
        mesh=mesh,
        runtime_sharding_resolver=_resolver(),
        kvdtype=jnp.float32,
        num_hidden_layers=1,
        num_kv_heads=1,
        max_model_length=32,
        head_dim=128,
        page_size=1,
    )


def _mla_config(mesh):
    return mla_cache_mod.MLARaggedPagesCacheConfig.create(
        mesh=mesh,
        runtime_sharding_resolver=_resolver(),
        kvdtype=jnp.bfloat16,
        num_hidden_layers=1,
        num_kv_heads=1,
        max_model_length=64,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        hbm_utilization=0.9,
        page_size=32,
    )


@pytest.mark.parametrize(
    "module, build",
    [(ragged_cache_mod, _ragged_config), (unified_cache_mod, _unified_config), (mla_cache_mod, _mla_config)],
)
def test_cache_configs_size_from_the_budget_of_their_own_mesh(monkeypatch, module, build):
    # A config that dropped its mesh would size from this process's budget alone, so hosts would disagree on the
    # page count again: the multi-host halt the agreement prevents.
    meshes = []

    def _budget(*_args, mesh=None, **_kwargs):
        meshes.append(mesh)
        return 1 << 26

    monkeypatch.setattr(module, "per_device_hbm_budget_bytes", _budget)
    mesh = _FakeMesh({"tp": 1})
    build(mesh)
    assert meshes and all(seen is mesh for seen in meshes)
