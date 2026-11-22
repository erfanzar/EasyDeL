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

"""Public cache-layout helpers resolve metadata without allocating arrays."""

import easydel.caching as caching
import easydel.caching.ragged_page as ragged_page
import easydel.caching.ragged_page.cache as cache_module
import pytest
from jax import numpy as jnp

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize(
    "name",
    ["canonicalize_dtype", "dtype_to_string", "resolve_ragged_cache_layout", "resolve_v3_kv_head_layout"],
)
def test_cache_layout_public_exports_preserve_legacy_imports(name):
    public = getattr(caching, name)
    assert public is getattr(ragged_page, name)
    assert public is getattr(cache_module, name)
    assert public is getattr(cache_module, "_" + name)


@pytest.mark.parametrize("dtype", ["float32", jnp.float32, jnp.dtype("float32")])
def test_cache_dtype_normalization_and_serialization(dtype):
    assert caching.canonicalize_dtype(dtype) == jnp.dtype("float32")
    assert caching.dtype_to_string(dtype) == "float32"


@pytest.mark.parametrize(
    "version,heads,shards,expected_shards",
    [("v2", 3, 4, 1), ("v2", 4, 4, 4), ("v3", 8, 4, 4), ("v3", 8, 1, 1)],
)
def test_public_ragged_layout_resolves_sharding(version, heads, shards, expected_shards):
    dtype, effective_shards = caching.resolve_ragged_cache_layout(
        "bfloat16", version=version, num_kv_heads=heads, k_headdim=128, kv_head_shards=shards
    )
    assert dtype == jnp.dtype("bfloat16")
    assert effective_shards == expected_shards


@pytest.mark.parametrize("head_dim,expected_shards,expected_heads", [(128, 4, 4), (64, 1, 2)])
def test_public_v3_layout_preserves_logical_heads_when_replicated(head_dim, expected_shards, expected_heads):
    dtype, shards, heads = caching.resolve_v3_kv_head_layout(
        "bfloat16", logical_num_kv_heads=2, k_headdim=head_dim, physical_kv_head_shards=4
    )
    assert dtype == jnp.dtype("bfloat16")
    assert shards == expected_shards
    assert heads == expected_heads
