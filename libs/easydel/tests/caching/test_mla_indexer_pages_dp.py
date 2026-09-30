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

"""Paged DSA indexer page allocation for single-replica and data-parallel serving.

Data-parallel serving builds rank-major inputs; the paged indexer runs once per DP rank on
that rank's page shard, so the pages must be allocated and sharded over the DP page axis.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.caching import MLARaggedPagesCacheConfig
from easydel.infra import EasyDeLBaseConfig

PAGE = 16


def _create(axis_dims, kvdtype=jnp.bfloat16):
    config = EasyDeLBaseConfig(sharding_axis_dims=axis_dims)
    return MLARaggedPagesCacheConfig.create(
        mesh=config.mesh,
        runtime_sharding_resolver=config.runtime_sharding_resolver,
        kvdtype=kvdtype,
        num_hidden_layers=2,
        num_kv_heads=4,
        max_model_length=256,
        kv_lora_rank=128,
        qk_rope_head_dim=0,
        page_size=PAGE,
        max_cache_tokens=64 * PAGE,
        indexer_key_dim=128,
        indexer_gate_dim=128,
        indexer_kpool=4,
    )


@pytest.mark.skipif(len(jax.devices()) < 2, reason="needs >= 2 devices for a dp=2 mesh")
def test_data_parallel_serving_keeps_dp_sharded_indexer_pages():
    """Rank-major DP serving runs the paged indexer per rank, so its pages must exist and be DP-sharded."""
    from easydel.caching import MLARaggedPagesCacheView

    axis_dims = (1, 2, -1, 1, 1, 1)
    cfg = _create(axis_dims)
    assert cfg.data_parallel_size == 2
    assert cfg.has_indexer_pages
    assert cfg.num_pages % 2 == 0
    token, pool = cfg.get_indexer_shapes()
    assert token == (cfg.num_pages, PAGE, 256)
    assert pool == (cfg.num_pages, PAGE // 4, 128)

    base = EasyDeLBaseConfig(sharding_axis_dims=axis_dims)
    view = MLARaggedPagesCacheView.init(
        cfg, layer_index=0, mesh=base.mesh, runtime_sharding_resolver=base.runtime_sharding_resolver
    )
    dp_sharding = view.indexer_dp_sharding()
    assert dp_sharding is not None
    mesh, names = dp_sharding
    assert int(np.prod([mesh.shape[n] for n in names])) == 2
    # Replicated-kernel dispatch is only for unsharded pages.
    assert view.indexer_kernel_mesh() is None


def test_single_replica_serving_keeps_indexer_pages():
    cfg = _create((1, 1, -1, 1, 1, 1))
    assert cfg.data_parallel_size == 1
    assert cfg.has_indexer_pages
    token, pool = cfg.get_indexer_shapes()
    assert token == (cfg.num_pages, PAGE, 256)
    assert pool == (cfg.num_pages, PAGE // 4, 128)


@pytest.mark.parametrize(
    ("kvdtype", "expected"),
    [
        (jnp.bfloat16, jnp.bfloat16),
        (jnp.float32, jnp.float32),
        # A quantized KV cache must not quantize the indexer keys that drive top-k selection.
        (jnp.float8_e4m3fn, jnp.bfloat16),
    ],
)
def test_indexer_dtype_defaults_to_kv_dtype_but_never_below_16_bits(kvdtype, expected):
    cfg = _create((1, 1, -1, 1, 1, 1), kvdtype=kvdtype)
    assert cfg.has_indexer_pages
    assert jnp.dtype(cfg.indexer_dtype) == jnp.dtype(expected)


def _create_owners(**owner_kwargs):
    config = EasyDeLBaseConfig(sharding_axis_dims=(1, 1, -1, 1, 1, 1))
    cfg = MLARaggedPagesCacheConfig.create(
        mesh=config.mesh,
        runtime_sharding_resolver=config.runtime_sharding_resolver,
        kvdtype=jnp.bfloat16,
        num_hidden_layers=2,
        num_kv_heads=4,
        max_model_length=256,
        kv_lora_rank=128,
        qk_rope_head_dim=0,
        page_size=PAGE,
        max_cache_tokens=64 * PAGE,
        indexer_key_dim=128,
        indexer_gate_dim=128,
        indexer_kpool=4,
        **owner_kwargs,
    )
    return config, cfg


def _owners(cache):
    return [view.indexer_token_pages is not None for view in cache.views]


def test_init_cache_allocates_exactly_the_budgeted_indexer_owners():
    """create() budgets ``indexer_num_layers`` owners; init_cache must not allocate more (r11 S09-1)."""
    from easydel.caching import MLARaggedPagesCache

    # Default: every layer is charged and every layer owns indexer pages.
    base, cfg = _create_owners()
    assert cfg.indexer_num_layers == 2
    cache = MLARaggedPagesCache.init_cache(base.mesh, cfg, base.runtime_sharding_resolver)
    assert _owners(cache) == [True, True]

    # Explicit schedule: only layer 1 owns pages, and the budget charges one owner.
    base, cfg = _create_owners(indexer_layer_indices=(1,))
    assert cfg.indexer_num_layers == 1 and cfg.indexer_layer_indices == (1,)
    cache = MLARaggedPagesCache.init_cache(base.mesh, cfg, base.runtime_sharding_resolver)
    assert _owners(cache) == [False, True]
    owner = cache.views[1]
    token, pool = cfg.get_indexer_shapes()
    assert owner.indexer_token_pages.shape == token and owner.indexer_pool_pages.shape == pool
    assert cache.views[0].indexer_pool_pages is None

    # A reduced count with no schedule cannot be allocated within budget: refuse instead of
    # silently allocating indexer pages on every layer.
    base, cfg = _create_owners(indexer_num_layers=1)
    with pytest.raises(ValueError, match="owner schedule"):
        MLARaggedPagesCache.init_cache(base.mesh, cfg, base.runtime_sharding_resolver)


def test_indexer_owner_schedule_is_validated():
    with pytest.raises(ValueError, match="must lie in"):
        _create_owners(indexer_layer_indices=(2,))
    with pytest.raises(ValueError, match="disagrees"):
        _create_owners(indexer_layer_indices=(0,), indexer_num_layers=2)
