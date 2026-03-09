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
import pytest

from easydel.caching.ragged_page.cache import (
    _canonicalize_dtype,
    _dtype_to_string,
    _select_compatible_v3_kv_cache_dtype,
)


def test_canonicalize_dtype_normalizes_class_and_instance():
    assert _canonicalize_dtype(jnp.bfloat16) == _canonicalize_dtype(jnp.dtype("bfloat16"))


def test_dtype_to_string_known():
    s = _dtype_to_string(jnp.bfloat16)
    assert isinstance(s, str) and len(s) > 0


def test_select_compatible_returns_original_when_sharding_is_valid():
    # 8 kv heads, headdim != 64 -> combined = align(8*2, packing).
    # With bf16 packing=1, groups=16.  16 % 4 == 0 -> ok.
    result = _select_compatible_v3_kv_cache_dtype(
        jnp.bfloat16, num_kv_heads=8, k_headdim=128, kv_head_shards=4
    )
    assert result == _canonicalize_dtype(jnp.bfloat16)


def test_select_compatible_returns_original_when_no_tp():
    result = _select_compatible_v3_kv_cache_dtype(
        jnp.float8_e4m3fn, num_kv_heads=3, k_headdim=128, kv_head_shards=1
    )
    assert result == _canonicalize_dtype(jnp.float8_e4m3fn)


def test_select_compatible_upcasts_when_groups_not_divisible():
    # Pick parameters where fp8 packing causes indivisible groups
    # but bf16 (packing=1) works.  num_kv_heads=4, headdim=128.
    # fp8 packing ~ 4 -> combined = align(8,4)=8, groups=8/4=2.
    # 2 % 4 != 0, so should upcast.
    # bf16 packing=1 -> combined = align(8,1)=8, groups=8/1=8.  8%4==0.
    result = _select_compatible_v3_kv_cache_dtype(
        jnp.float8_e4m3fn, num_kv_heads=4, k_headdim=128, kv_head_shards=4
    )
    assert result != _canonicalize_dtype(jnp.float8_e4m3fn)


def test_select_compatible_raises_when_nothing_works():
    # 1 kv head, headdim=128 -> combined heads always small.
    # With kv_head_shards very large, nothing divides.
    with pytest.raises(ValueError, match="incompatible"):
        _select_compatible_v3_kv_cache_dtype(
            jnp.float8_e4m3fn, num_kv_heads=1, k_headdim=128, kv_head_shards=1024
        )
